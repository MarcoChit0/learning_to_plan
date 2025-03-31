; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 572451

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child368 - child
    bread57 - bread-portion
    content204 - content-portion
    tray346 tray154 - tray
    table378 table231 table285 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray346 kitchen)
     (at tray154 kitchen)
     (at_kitchen_bread bread57)
     (at_kitchen_content content204)
     (not_allergic_gluten child368)
     (waiting child368 table285)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child368)
    )
  )
)
