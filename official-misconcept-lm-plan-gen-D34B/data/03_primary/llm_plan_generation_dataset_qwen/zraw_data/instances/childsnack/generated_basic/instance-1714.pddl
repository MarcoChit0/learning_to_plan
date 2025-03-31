; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 102743

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child468 - child
    bread348 - bread-portion
    content36 - content-portion
    tray226 tray287 tray446 - tray
    table91 table138 table76 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray226 kitchen)
     (at tray287 kitchen)
     (at tray446 kitchen)
     (at_kitchen_bread bread348)
     (at_kitchen_content content36)
     (not_allergic_gluten child468)
     (waiting child468 table91)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child468)
    )
  )
)
