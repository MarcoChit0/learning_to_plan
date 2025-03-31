; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 733592

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child34 - child
    bread77 - bread-portion
    content184 - content-portion
    tray150 tray464 tray172 - tray
    table57 table215 table65 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray150 kitchen)
     (at tray464 kitchen)
     (at tray172 kitchen)
     (at_kitchen_bread bread77)
     (at_kitchen_content content184)
     (not_allergic_gluten child34)
     (waiting child34 table57)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child34)
    )
  )
)
