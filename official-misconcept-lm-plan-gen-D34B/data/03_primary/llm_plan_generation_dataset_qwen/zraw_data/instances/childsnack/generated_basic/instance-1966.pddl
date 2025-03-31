; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 264370

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child106 child67 - child
    bread313 bread74 - bread-portion
    content176 content47 - content-portion
    tray68 - tray
    table152 table258 table113 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray68 kitchen)
     (at_kitchen_bread bread313)
     (at_kitchen_bread bread74)
     (at_kitchen_content content176)
     (at_kitchen_content content47)
     (no_gluten_bread bread313)
     (no_gluten_content content176)
     (allergic_gluten child67)
     (not_allergic_gluten child106)
     (waiting child106 table113)
     (waiting child67 table258)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child106)
     (served child67)
    )
  )
)
