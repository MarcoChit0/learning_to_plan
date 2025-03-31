; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 145381

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child132 - child
    bread173 - bread-portion
    content424 - content-portion
    tray464 tray114 tray213 tray61 - tray
    table117 table413 table472 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray464 kitchen)
     (at tray114 kitchen)
     (at tray213 kitchen)
     (at tray61 kitchen)
     (at_kitchen_bread bread173)
     (at_kitchen_content content424)
     (not_allergic_gluten child132)
     (waiting child132 table117)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child132)
    )
  )
)
