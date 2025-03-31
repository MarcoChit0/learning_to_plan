; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 365054

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child289 - child
    bread15 - bread-portion
    content326 - content-portion
    tray32 tray115 tray397 tray106 - tray
    table326 table166 table144 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray32 kitchen)
     (at tray115 kitchen)
     (at tray397 kitchen)
     (at tray106 kitchen)
     (at_kitchen_bread bread15)
     (at_kitchen_content content326)
     (not_allergic_gluten child289)
     (waiting child289 table144)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child289)
    )
  )
)
