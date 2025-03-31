; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 288712

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child396 - child
    bread167 - bread-portion
    content150 - content-portion
    tray255 - tray
    table463 table239 table179 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray255 kitchen)
     (at_kitchen_bread bread167)
     (at_kitchen_content content150)
     (not_allergic_gluten child396)
     (waiting child396 table463)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child396)
    )
  )
)
