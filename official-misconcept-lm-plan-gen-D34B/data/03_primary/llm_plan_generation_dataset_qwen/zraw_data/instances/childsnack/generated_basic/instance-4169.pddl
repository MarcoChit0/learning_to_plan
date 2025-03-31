; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 812929

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child466 child249 - child
    bread368 bread357 - bread-portion
    content323 content270 - content-portion
    tray132 - tray
    table95 table36 table391 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray132 kitchen)
     (at_kitchen_bread bread368)
     (at_kitchen_bread bread357)
     (at_kitchen_content content323)
     (at_kitchen_content content270)
     (not_allergic_gluten child249)
     (not_allergic_gluten child466)
     (waiting child466 table36)
     (waiting child249 table391)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child466)
     (served child249)
    )
  )
)
