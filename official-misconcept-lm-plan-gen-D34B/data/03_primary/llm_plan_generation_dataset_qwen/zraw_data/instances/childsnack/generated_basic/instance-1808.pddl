; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 735098

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child352 child411 - child
    bread316 bread487 - bread-portion
    content80 content484 - content-portion
    tray125 - tray
    table269 table416 table331 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray125 kitchen)
     (at_kitchen_bread bread316)
     (at_kitchen_bread bread487)
     (at_kitchen_content content80)
     (at_kitchen_content content484)
     (no_gluten_bread bread487)
     (no_gluten_content content80)
     (allergic_gluten child411)
     (not_allergic_gluten child352)
     (waiting child352 table331)
     (waiting child411 table416)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child352)
     (served child411)
    )
  )
)
