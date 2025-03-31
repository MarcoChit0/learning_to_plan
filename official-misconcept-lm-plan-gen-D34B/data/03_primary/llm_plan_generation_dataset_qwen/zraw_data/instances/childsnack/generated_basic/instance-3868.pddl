; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 328890

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child185 - child
    bread287 - bread-portion
    content299 - content-portion
    tray340 tray239 - tray
    table321 table436 table5 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray340 kitchen)
     (at tray239 kitchen)
     (at_kitchen_bread bread287)
     (at_kitchen_content content299)
     (not_allergic_gluten child185)
     (waiting child185 table436)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child185)
    )
  )
)
