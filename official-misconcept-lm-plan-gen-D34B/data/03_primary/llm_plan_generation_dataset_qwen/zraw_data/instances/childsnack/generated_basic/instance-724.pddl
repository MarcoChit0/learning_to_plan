; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 437074

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child345 - child
    bread259 - bread-portion
    content200 - content-portion
    tray155 tray469 tray159 tray439 - tray
    table460 table388 table248 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray155 kitchen)
     (at tray469 kitchen)
     (at tray159 kitchen)
     (at tray439 kitchen)
     (at_kitchen_bread bread259)
     (at_kitchen_content content200)
     (not_allergic_gluten child345)
     (waiting child345 table388)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child345)
    )
  )
)
