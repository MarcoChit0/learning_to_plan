; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 380567

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child475 - child
    bread492 - bread-portion
    content491 - content-portion
    tray418 - tray
    table153 table21 table117 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray418 kitchen)
     (at_kitchen_bread bread492)
     (at_kitchen_content content491)
     (not_allergic_gluten child475)
     (waiting child475 table117)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child475)
    )
  )
)
