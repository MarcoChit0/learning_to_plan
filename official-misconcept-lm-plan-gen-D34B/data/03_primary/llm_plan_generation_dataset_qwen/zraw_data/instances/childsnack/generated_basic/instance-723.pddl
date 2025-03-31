; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 147490

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child453 - child
    bread12 - bread-portion
    content446 - content-portion
    tray219 tray240 tray93 tray1 - tray
    table167 table17 table388 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray219 kitchen)
     (at tray240 kitchen)
     (at tray93 kitchen)
     (at tray1 kitchen)
     (at_kitchen_bread bread12)
     (at_kitchen_content content446)
     (not_allergic_gluten child453)
     (waiting child453 table167)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child453)
    )
  )
)
