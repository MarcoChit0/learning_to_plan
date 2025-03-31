; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 910364

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child450 child493 - child
    bread13 bread374 - bread-portion
    content58 content23 - content-portion
    tray273 tray13 tray429 tray230 - tray
    table231 table375 table20 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray273 kitchen)
     (at tray13 kitchen)
     (at tray429 kitchen)
     (at tray230 kitchen)
     (at_kitchen_bread bread13)
     (at_kitchen_bread bread374)
     (at_kitchen_content content58)
     (at_kitchen_content content23)
     (not_allergic_gluten child493)
     (not_allergic_gluten child450)
     (waiting child450 table375)
     (waiting child493 table375)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child450)
     (served child493)
    )
  )
)
