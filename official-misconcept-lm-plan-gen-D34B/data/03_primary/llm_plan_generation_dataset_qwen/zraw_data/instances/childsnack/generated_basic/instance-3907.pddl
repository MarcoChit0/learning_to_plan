; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 523323

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child267 - child
    bread235 - bread-portion
    content356 - content-portion
    tray204 tray65 - tray
    table483 table104 table320 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray204 kitchen)
     (at tray65 kitchen)
     (at_kitchen_bread bread235)
     (at_kitchen_content content356)
     (not_allergic_gluten child267)
     (waiting child267 table320)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child267)
    )
  )
)
