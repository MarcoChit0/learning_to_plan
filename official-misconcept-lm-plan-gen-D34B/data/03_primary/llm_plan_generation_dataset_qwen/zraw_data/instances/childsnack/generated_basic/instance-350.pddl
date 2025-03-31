; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 524291

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child302 - child
    bread293 - bread-portion
    content226 - content-portion
    tray418 tray127 tray286 tray419 - tray
    table105 table47 table271 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray418 kitchen)
     (at tray127 kitchen)
     (at tray286 kitchen)
     (at tray419 kitchen)
     (at_kitchen_bread bread293)
     (at_kitchen_content content226)
     (not_allergic_gluten child302)
     (waiting child302 table105)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child302)
    )
  )
)
