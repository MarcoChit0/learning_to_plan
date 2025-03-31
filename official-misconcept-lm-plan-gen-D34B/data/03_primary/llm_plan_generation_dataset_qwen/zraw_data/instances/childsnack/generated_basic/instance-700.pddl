; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 841747

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child238 - child
    bread264 - bread-portion
    content350 - content-portion
    tray376 tray25 tray415 tray241 - tray
    table57 table400 table486 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray376 kitchen)
     (at tray25 kitchen)
     (at tray415 kitchen)
     (at tray241 kitchen)
     (at_kitchen_bread bread264)
     (at_kitchen_content content350)
     (not_allergic_gluten child238)
     (waiting child238 table57)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child238)
    )
  )
)
