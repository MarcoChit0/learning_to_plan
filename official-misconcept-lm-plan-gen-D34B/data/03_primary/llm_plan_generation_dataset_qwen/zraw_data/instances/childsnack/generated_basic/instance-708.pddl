; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 207993

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child257 - child
    bread3 - bread-portion
    content317 - content-portion
    tray342 tray427 tray489 tray216 - tray
    table289 table342 table61 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray342 kitchen)
     (at tray427 kitchen)
     (at tray489 kitchen)
     (at tray216 kitchen)
     (at_kitchen_bread bread3)
     (at_kitchen_content content317)
     (not_allergic_gluten child257)
     (waiting child257 table61)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child257)
    )
  )
)
