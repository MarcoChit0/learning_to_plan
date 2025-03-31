; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 393979

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child281 - child
    bread116 - bread-portion
    content268 - content-portion
    tray478 tray425 tray183 tray122 - tray
    table221 table91 table194 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray478 kitchen)
     (at tray425 kitchen)
     (at tray183 kitchen)
     (at tray122 kitchen)
     (at_kitchen_bread bread116)
     (at_kitchen_content content268)
     (not_allergic_gluten child281)
     (waiting child281 table91)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child281)
    )
  )
)
