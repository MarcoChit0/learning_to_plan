; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 109200

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child103 - child
    bread188 - bread-portion
    content352 - content-portion
    tray96 tray55 tray421 tray89 - tray
    table421 table448 table486 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray96 kitchen)
     (at tray55 kitchen)
     (at tray421 kitchen)
     (at tray89 kitchen)
     (at_kitchen_bread bread188)
     (at_kitchen_content content352)
     (not_allergic_gluten child103)
     (waiting child103 table448)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child103)
    )
  )
)
