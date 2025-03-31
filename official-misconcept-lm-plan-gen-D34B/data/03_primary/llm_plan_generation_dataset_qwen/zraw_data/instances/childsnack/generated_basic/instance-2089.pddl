; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 434574

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child9 - child
    bread114 - bread-portion
    content378 - content-portion
    tray95 tray123 tray257 tray489 - tray
    table426 table330 table183 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray95 kitchen)
     (at tray123 kitchen)
     (at tray257 kitchen)
     (at tray489 kitchen)
     (at_kitchen_bread bread114)
     (at_kitchen_content content378)
     (not_allergic_gluten child9)
     (waiting child9 table330)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child9)
    )
  )
)
