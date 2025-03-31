; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 335800

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child290 child465 - child
    bread413 bread327 - bread-portion
    content47 content32 - content-portion
    tray310 - tray
    table349 table214 table360 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray310 kitchen)
     (at_kitchen_bread bread413)
     (at_kitchen_bread bread327)
     (at_kitchen_content content47)
     (at_kitchen_content content32)
     (no_gluten_bread bread327)
     (no_gluten_content content32)
     (allergic_gluten child290)
     (not_allergic_gluten child465)
     (waiting child290 table214)
     (waiting child465 table360)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child290)
     (served child465)
    )
  )
)
