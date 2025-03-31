; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 998878

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child194 child101 - child
    bread58 bread425 - bread-portion
    content113 content55 - content-portion
    tray59 tray209 tray284 - tray
    table264 table418 table259 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray59 kitchen)
     (at tray209 kitchen)
     (at tray284 kitchen)
     (at_kitchen_bread bread58)
     (at_kitchen_bread bread425)
     (at_kitchen_content content113)
     (at_kitchen_content content55)
     (no_gluten_bread bread425)
     (no_gluten_content content55)
     (allergic_gluten child101)
     (not_allergic_gluten child194)
     (waiting child194 table264)
     (waiting child101 table259)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child194)
     (served child101)
    )
  )
)
