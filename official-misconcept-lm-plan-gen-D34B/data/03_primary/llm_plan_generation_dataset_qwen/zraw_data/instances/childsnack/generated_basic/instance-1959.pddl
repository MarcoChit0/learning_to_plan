; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 879687

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child96 child328 - child
    bread368 bread187 - bread-portion
    content292 content43 - content-portion
    tray87 - tray
    table347 table39 table35 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray87 kitchen)
     (at_kitchen_bread bread368)
     (at_kitchen_bread bread187)
     (at_kitchen_content content292)
     (at_kitchen_content content43)
     (no_gluten_bread bread187)
     (no_gluten_content content43)
     (allergic_gluten child328)
     (not_allergic_gluten child96)
     (waiting child96 table347)
     (waiting child328 table35)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child96)
     (served child328)
    )
  )
)
