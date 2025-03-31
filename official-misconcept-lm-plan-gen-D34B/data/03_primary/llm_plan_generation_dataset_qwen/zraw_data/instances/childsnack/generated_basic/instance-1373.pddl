; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 15092

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child354 child156 - child
    bread165 bread449 - bread-portion
    content21 content68 - content-portion
    tray231 tray36 tray394 - tray
    table193 table37 table6 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray231 kitchen)
     (at tray36 kitchen)
     (at tray394 kitchen)
     (at_kitchen_bread bread165)
     (at_kitchen_bread bread449)
     (at_kitchen_content content21)
     (at_kitchen_content content68)
     (not_allergic_gluten child354)
     (not_allergic_gluten child156)
     (waiting child354 table37)
     (waiting child156 table193)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child354)
     (served child156)
    )
  )
)
