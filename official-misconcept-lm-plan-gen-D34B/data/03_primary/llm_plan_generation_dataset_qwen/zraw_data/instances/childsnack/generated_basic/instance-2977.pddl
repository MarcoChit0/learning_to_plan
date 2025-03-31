; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 486574

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child493 - child
    bread120 - bread-portion
    content176 - content-portion
    tray489 tray291 tray409 tray278 - tray
    table81 table278 table332 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray489 kitchen)
     (at tray291 kitchen)
     (at tray409 kitchen)
     (at tray278 kitchen)
     (at_kitchen_bread bread120)
     (at_kitchen_content content176)
     (not_allergic_gluten child493)
     (waiting child493 table81)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child493)
    )
  )
)
