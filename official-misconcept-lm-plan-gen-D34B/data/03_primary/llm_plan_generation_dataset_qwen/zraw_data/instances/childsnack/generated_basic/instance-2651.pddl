; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 975846

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child223 - child
    bread113 - bread-portion
    content78 - content-portion
    tray382 tray457 tray143 tray362 - tray
    table492 table365 table43 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray382 kitchen)
     (at tray457 kitchen)
     (at tray143 kitchen)
     (at tray362 kitchen)
     (at_kitchen_bread bread113)
     (at_kitchen_content content78)
     (not_allergic_gluten child223)
     (waiting child223 table365)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child223)
    )
  )
)
