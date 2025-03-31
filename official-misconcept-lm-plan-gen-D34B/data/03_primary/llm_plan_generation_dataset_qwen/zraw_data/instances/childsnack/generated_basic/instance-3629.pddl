; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 987965

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child10 child13 - child
    bread287 bread90 - bread-portion
    content447 content369 - content-portion
    tray142 tray313 tray262 tray77 - tray
    table197 table158 table425 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray142 kitchen)
     (at tray313 kitchen)
     (at tray262 kitchen)
     (at tray77 kitchen)
     (at_kitchen_bread bread287)
     (at_kitchen_bread bread90)
     (at_kitchen_content content447)
     (at_kitchen_content content369)
     (no_gluten_bread bread90)
     (no_gluten_content content447)
     (allergic_gluten child10)
     (not_allergic_gluten child13)
     (waiting child10 table197)
     (waiting child13 table197)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child10)
     (served child13)
    )
  )
)
