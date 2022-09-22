# ==================================================================
# logging functions
# ------------------------------------------------------------------

LogTime()
{
  # Usage:
  # LogTime [message]
  echo -e $(date +"%Y/%m/%d %H:%M:%S") - $1
}

LogElapsedTime()
{
  # Usage:
  # start SECONDS variable, then call function
  # SECONDS=0
  # LogElapsedTime [SECONDS] [name] [send_telegram_msg]
  days=$(($1 / (3600 * 24)))
  hours=$((($1 % (3600 * 24)) / 3600))
  mins=$(((($1 % (3600 * 24)) % 3600) / 60))
  secs=$(((($1 % (3600 * 24)) % 3600) % 60))
  msg="Finished running $2 in"

  if [ $days -gt 1 ] ; then
    msg+=" $days days,"
  fi

  if [ $hours -gt 1 ] ; then
    msg+=" $hours hours,"
  fi
  msg+=" $mins mins and $secs secs on $HOSTNAME."

  if [ "$#" -eq 3 ] && [ $3 -eq 1 ] ; then
    telegram-send "$msg"
  fi

  echo $msg
}
