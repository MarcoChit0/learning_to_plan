; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 365371

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child170 - child
    bread386 - bread-portion
    content190 - content-portion
    tray188 tray224 tray254 - tray
    table301 table497 table290 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray188 kitchen)
     (at tray224 kitchen)
     (at tray254 kitchen)
     (at_kitchen_bread bread386)
     (at_kitchen_content content190)
     (not_allergic_gluten child170)
     (waiting child170 table301)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child170)
    )
  )
)
