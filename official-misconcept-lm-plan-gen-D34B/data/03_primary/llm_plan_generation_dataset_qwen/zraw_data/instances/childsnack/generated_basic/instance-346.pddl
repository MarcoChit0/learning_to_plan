; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 115778

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child441 - child
    bread407 - bread-portion
    content219 - content-portion
    tray141 tray210 tray364 tray112 - tray
    table150 table437 table413 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray141 kitchen)
     (at tray210 kitchen)
     (at tray364 kitchen)
     (at tray112 kitchen)
     (at_kitchen_bread bread407)
     (at_kitchen_content content219)
     (not_allergic_gluten child441)
     (waiting child441 table437)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child441)
    )
  )
)
