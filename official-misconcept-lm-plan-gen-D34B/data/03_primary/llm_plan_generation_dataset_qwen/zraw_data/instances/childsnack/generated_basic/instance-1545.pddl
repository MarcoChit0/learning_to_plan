; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 861361

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child355 child252 - child
    bread226 bread424 - bread-portion
    content386 content55 - content-portion
    tray49 tray476 tray335 tray407 - tray
    table316 table365 table379 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray49 kitchen)
     (at tray476 kitchen)
     (at tray335 kitchen)
     (at tray407 kitchen)
     (at_kitchen_bread bread226)
     (at_kitchen_bread bread424)
     (at_kitchen_content content386)
     (at_kitchen_content content55)
     (not_allergic_gluten child252)
     (not_allergic_gluten child355)
     (waiting child355 table379)
     (waiting child252 table316)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child355)
     (served child252)
    )
  )
)
