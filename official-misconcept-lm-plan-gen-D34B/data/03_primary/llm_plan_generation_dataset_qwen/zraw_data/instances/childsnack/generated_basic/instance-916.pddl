; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 93470

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child224 child276 child397 - child
    bread166 bread323 bread303 - bread-portion
    content426 content242 content338 - content-portion
    tray376 - tray
    table77 table421 table404 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray376 kitchen)
     (at_kitchen_bread bread166)
     (at_kitchen_bread bread323)
     (at_kitchen_bread bread303)
     (at_kitchen_content content426)
     (at_kitchen_content content242)
     (at_kitchen_content content338)
     (not_allergic_gluten child397)
     (not_allergic_gluten child276)
     (not_allergic_gluten child224)
     (waiting child224 table404)
     (waiting child276 table404)
     (waiting child397 table404)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child224)
     (served child276)
     (served child397)
    )
  )
)
