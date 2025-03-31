; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 208092

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child15 child375 - child
    bread259 bread9 - bread-portion
    content312 content111 - content-portion
    tray488 tray120 - tray
    table245 table98 table114 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray488 kitchen)
     (at tray120 kitchen)
     (at_kitchen_bread bread259)
     (at_kitchen_bread bread9)
     (at_kitchen_content content312)
     (at_kitchen_content content111)
     (not_allergic_gluten child15)
     (not_allergic_gluten child375)
     (waiting child15 table114)
     (waiting child375 table98)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child15)
     (served child375)
    )
  )
)
