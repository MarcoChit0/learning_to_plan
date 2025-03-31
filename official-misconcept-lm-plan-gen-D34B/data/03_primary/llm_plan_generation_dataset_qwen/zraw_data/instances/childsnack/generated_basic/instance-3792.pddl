; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 480090

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child379 - child
    bread152 - bread-portion
    content498 - content-portion
    tray189 - tray
    table358 table87 table230 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray189 kitchen)
     (at_kitchen_bread bread152)
     (at_kitchen_content content498)
     (not_allergic_gluten child379)
     (waiting child379 table358)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child379)
    )
  )
)
