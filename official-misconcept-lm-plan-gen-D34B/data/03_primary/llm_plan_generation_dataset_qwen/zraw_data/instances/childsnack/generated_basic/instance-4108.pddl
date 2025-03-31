; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 125105

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child276 - child
    bread464 - bread-portion
    content160 - content-portion
    tray115 tray466 tray82 - tray
    table356 table459 table47 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray115 kitchen)
     (at tray466 kitchen)
     (at tray82 kitchen)
     (at_kitchen_bread bread464)
     (at_kitchen_content content160)
     (not_allergic_gluten child276)
     (waiting child276 table459)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child276)
    )
  )
)
