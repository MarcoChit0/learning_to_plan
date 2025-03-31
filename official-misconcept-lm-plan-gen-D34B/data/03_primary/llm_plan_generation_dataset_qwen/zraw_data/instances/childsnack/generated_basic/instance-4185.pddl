; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 893393

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child388 child162 - child
    bread341 bread85 - bread-portion
    content384 content298 - content-portion
    tray31 - tray
    table239 table121 table341 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray31 kitchen)
     (at_kitchen_bread bread341)
     (at_kitchen_bread bread85)
     (at_kitchen_content content384)
     (at_kitchen_content content298)
     (not_allergic_gluten child162)
     (not_allergic_gluten child388)
     (waiting child388 table341)
     (waiting child162 table121)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child388)
     (served child162)
    )
  )
)
