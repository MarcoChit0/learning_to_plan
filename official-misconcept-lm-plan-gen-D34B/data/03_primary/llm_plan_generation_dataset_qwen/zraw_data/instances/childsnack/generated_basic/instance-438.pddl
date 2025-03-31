; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 230946

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child274 child191 child115 - child
    bread76 bread241 bread264 - bread-portion
    content483 content17 content425 - content-portion
    tray192 - tray
    table170 table90 table325 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray192 kitchen)
     (at_kitchen_bread bread76)
     (at_kitchen_bread bread241)
     (at_kitchen_bread bread264)
     (at_kitchen_content content483)
     (at_kitchen_content content17)
     (at_kitchen_content content425)
     (not_allergic_gluten child191)
     (not_allergic_gluten child274)
     (not_allergic_gluten child115)
     (waiting child274 table325)
     (waiting child191 table170)
     (waiting child115 table170)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child274)
     (served child191)
     (served child115)
    )
  )
)
