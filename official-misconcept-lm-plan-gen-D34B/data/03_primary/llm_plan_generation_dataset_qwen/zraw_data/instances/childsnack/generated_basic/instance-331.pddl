; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 876354

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child97 - child
    bread359 - bread-portion
    content74 - content-portion
    tray492 tray360 tray43 tray103 - tray
    table171 table318 table215 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray492 kitchen)
     (at tray360 kitchen)
     (at tray43 kitchen)
     (at tray103 kitchen)
     (at_kitchen_bread bread359)
     (at_kitchen_content content74)
     (not_allergic_gluten child97)
     (waiting child97 table318)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child97)
    )
  )
)
