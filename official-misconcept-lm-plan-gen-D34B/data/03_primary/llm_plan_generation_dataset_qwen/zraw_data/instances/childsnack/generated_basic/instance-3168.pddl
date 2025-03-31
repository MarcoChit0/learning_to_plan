; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 661931

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child425 child375 - child
    bread309 bread247 - bread-portion
    content372 content317 - content-portion
    tray434 tray377 - tray
    table469 table70 table405 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray434 kitchen)
     (at tray377 kitchen)
     (at_kitchen_bread bread309)
     (at_kitchen_bread bread247)
     (at_kitchen_content content372)
     (at_kitchen_content content317)
     (not_allergic_gluten child375)
     (not_allergic_gluten child425)
     (waiting child425 table405)
     (waiting child375 table469)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child425)
     (served child375)
    )
  )
)
