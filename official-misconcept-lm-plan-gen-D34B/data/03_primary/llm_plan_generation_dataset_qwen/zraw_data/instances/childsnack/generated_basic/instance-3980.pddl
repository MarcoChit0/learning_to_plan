; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 532845

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child368 - child
    bread465 - bread-portion
    content210 - content-portion
    tray159 tray470 tray200 - tray
    table50 table165 table76 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray159 kitchen)
     (at tray470 kitchen)
     (at tray200 kitchen)
     (at_kitchen_bread bread465)
     (at_kitchen_content content210)
     (not_allergic_gluten child368)
     (waiting child368 table50)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child368)
    )
  )
)
