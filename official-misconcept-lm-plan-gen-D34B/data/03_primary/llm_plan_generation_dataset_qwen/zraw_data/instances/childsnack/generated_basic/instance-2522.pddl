; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 438895

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child96 - child
    bread58 - bread-portion
    content107 - content-portion
    tray209 tray193 tray188 - tray
    table84 table408 table473 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray209 kitchen)
     (at tray193 kitchen)
     (at tray188 kitchen)
     (at_kitchen_bread bread58)
     (at_kitchen_content content107)
     (not_allergic_gluten child96)
     (waiting child96 table473)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child96)
    )
  )
)
