; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 744170

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child236 child165 - child
    bread142 bread61 - bread-portion
    content3 content386 - content-portion
    tray390 - tray
    table430 table114 table195 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray390 kitchen)
     (at_kitchen_bread bread142)
     (at_kitchen_bread bread61)
     (at_kitchen_content content3)
     (at_kitchen_content content386)
     (not_allergic_gluten child236)
     (not_allergic_gluten child165)
     (waiting child236 table114)
     (waiting child165 table430)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child236)
     (served child165)
    )
  )
)
