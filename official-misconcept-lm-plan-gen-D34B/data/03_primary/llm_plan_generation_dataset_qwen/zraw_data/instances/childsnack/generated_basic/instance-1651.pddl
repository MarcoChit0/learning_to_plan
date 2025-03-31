; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 360957

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child208 - child
    bread173 - bread-portion
    content205 - content-portion
    tray264 tray364 tray95 tray253 - tray
    table397 table164 table163 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray264 kitchen)
     (at tray364 kitchen)
     (at tray95 kitchen)
     (at tray253 kitchen)
     (at_kitchen_bread bread173)
     (at_kitchen_content content205)
     (not_allergic_gluten child208)
     (waiting child208 table164)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child208)
    )
  )
)
