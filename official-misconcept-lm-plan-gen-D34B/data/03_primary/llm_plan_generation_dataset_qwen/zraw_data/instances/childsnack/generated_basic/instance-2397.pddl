; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 763696

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child300 child38 - child
    bread229 bread461 - bread-portion
    content103 content300 - content-portion
    tray108 - tray
    table215 table203 table47 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray108 kitchen)
     (at_kitchen_bread bread229)
     (at_kitchen_bread bread461)
     (at_kitchen_content content103)
     (at_kitchen_content content300)
     (not_allergic_gluten child38)
     (not_allergic_gluten child300)
     (waiting child300 table47)
     (waiting child38 table215)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child300)
     (served child38)
    )
  )
)
