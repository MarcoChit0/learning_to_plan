; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 394417

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child494 child309 - child
    bread363 bread371 - bread-portion
    content460 content174 - content-portion
    tray300 - tray
    table117 table208 table327 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray300 kitchen)
     (at_kitchen_bread bread363)
     (at_kitchen_bread bread371)
     (at_kitchen_content content460)
     (at_kitchen_content content174)
     (not_allergic_gluten child309)
     (not_allergic_gluten child494)
     (waiting child494 table117)
     (waiting child309 table117)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child494)
     (served child309)
    )
  )
)
