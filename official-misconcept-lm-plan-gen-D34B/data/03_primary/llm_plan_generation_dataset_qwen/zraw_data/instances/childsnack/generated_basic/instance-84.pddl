; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 37984

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child82 child457 - child
    bread239 bread364 - bread-portion
    content482 content226 - content-portion
    tray165 tray269 - tray
    table156 table142 table417 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray165 kitchen)
     (at tray269 kitchen)
     (at_kitchen_bread bread239)
     (at_kitchen_bread bread364)
     (at_kitchen_content content482)
     (at_kitchen_content content226)
     (not_allergic_gluten child82)
     (not_allergic_gluten child457)
     (waiting child82 table142)
     (waiting child457 table142)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child82)
     (served child457)
    )
  )
)
