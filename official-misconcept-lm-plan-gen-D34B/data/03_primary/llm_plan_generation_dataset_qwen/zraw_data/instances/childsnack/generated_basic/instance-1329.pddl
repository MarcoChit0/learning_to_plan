; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 46256

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child102 child375 - child
    bread113 bread454 - bread-portion
    content60 content307 - content-portion
    tray40 tray202 tray161 - tray
    table303 table258 table356 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray40 kitchen)
     (at tray202 kitchen)
     (at tray161 kitchen)
     (at_kitchen_bread bread113)
     (at_kitchen_bread bread454)
     (at_kitchen_content content60)
     (at_kitchen_content content307)
     (not_allergic_gluten child102)
     (not_allergic_gluten child375)
     (waiting child102 table258)
     (waiting child375 table356)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child102)
     (served child375)
    )
  )
)
