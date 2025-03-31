; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 197757

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child164 child418 child431 - child
    bread339 bread232 bread434 - bread-portion
    content306 content259 content407 - content-portion
    tray226 - tray
    table119 table425 table12 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray226 kitchen)
     (at_kitchen_bread bread339)
     (at_kitchen_bread bread232)
     (at_kitchen_bread bread434)
     (at_kitchen_content content306)
     (at_kitchen_content content259)
     (at_kitchen_content content407)
     (not_allergic_gluten child164)
     (not_allergic_gluten child431)
     (not_allergic_gluten child418)
     (waiting child164 table12)
     (waiting child418 table119)
     (waiting child431 table119)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child164)
     (served child418)
     (served child431)
    )
  )
)
