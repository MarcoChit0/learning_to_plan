; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 681826

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child490 child114 - child
    bread484 bread481 - bread-portion
    content147 content65 - content-portion
    tray104 tray181 - tray
    table55 table308 table263 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray104 kitchen)
     (at tray181 kitchen)
     (at_kitchen_bread bread484)
     (at_kitchen_bread bread481)
     (at_kitchen_content content147)
     (at_kitchen_content content65)
     (not_allergic_gluten child490)
     (not_allergic_gluten child114)
     (waiting child490 table308)
     (waiting child114 table308)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child490)
     (served child114)
    )
  )
)
