; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 257850

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child206 - child
    bread475 - bread-portion
    content483 - content-portion
    tray478 tray117 tray340 - tray
    table185 table15 table292 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray478 kitchen)
     (at tray117 kitchen)
     (at tray340 kitchen)
     (at_kitchen_bread bread475)
     (at_kitchen_content content483)
     (not_allergic_gluten child206)
     (waiting child206 table15)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child206)
    )
  )
)
