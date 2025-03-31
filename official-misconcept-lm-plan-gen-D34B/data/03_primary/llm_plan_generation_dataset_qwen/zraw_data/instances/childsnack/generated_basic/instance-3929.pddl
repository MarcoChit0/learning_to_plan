; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 436045

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child478 child13 - child
    bread487 bread455 - bread-portion
    content377 content70 - content-portion
    tray155 - tray
    table92 table99 table217 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray155 kitchen)
     (at_kitchen_bread bread487)
     (at_kitchen_bread bread455)
     (at_kitchen_content content377)
     (at_kitchen_content content70)
     (no_gluten_bread bread455)
     (no_gluten_content content377)
     (allergic_gluten child13)
     (not_allergic_gluten child478)
     (waiting child478 table217)
     (waiting child13 table217)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child478)
     (served child13)
    )
  )
)
