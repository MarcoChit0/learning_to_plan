; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 999637

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child85 - child
    bread306 - bread-portion
    content403 - content-portion
    tray267 tray276 tray273 tray151 - tray
    table273 table254 table34 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray267 kitchen)
     (at tray276 kitchen)
     (at tray273 kitchen)
     (at tray151 kitchen)
     (at_kitchen_bread bread306)
     (at_kitchen_content content403)
     (not_allergic_gluten child85)
     (waiting child85 table254)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child85)
    )
  )
)
