; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 800299

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child424 - child
    bread492 - bread-portion
    content55 - content-portion
    tray177 tray235 tray460 tray429 - tray
    table127 table161 table308 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray177 kitchen)
     (at tray235 kitchen)
     (at tray460 kitchen)
     (at tray429 kitchen)
     (at_kitchen_bread bread492)
     (at_kitchen_content content55)
     (not_allergic_gluten child424)
     (waiting child424 table308)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child424)
    )
  )
)
