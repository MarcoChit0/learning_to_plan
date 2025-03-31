; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 790962

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child219 - child
    bread341 - bread-portion
    content485 - content-portion
    tray193 tray24 tray330 tray394 - tray
    table305 table18 table349 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray193 kitchen)
     (at tray24 kitchen)
     (at tray330 kitchen)
     (at tray394 kitchen)
     (at_kitchen_bread bread341)
     (at_kitchen_content content485)
     (not_allergic_gluten child219)
     (waiting child219 table305)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child219)
    )
  )
)
